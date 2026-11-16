//! Linux: crash-handler's signal handler kept inside the signal's frame (#1239).
//!
//! crash-handler copies the context a crash signal comes with as glibc's `ucontext_t` (968 bytes
//! on x86_64). The kernel's frame holds less: its own `ucontext` (without glibc's room for the
//! floating-point registers and the shadow stack), the signal's info, then the processor's state,
//! up to the top of the alternate signal stack the handler runs on. Where that state is large (a
//! processor with XSAVE: AVX and later) the copy stays inside it. Without XSAVE (an older
//! processor, or a virtual machine that doesn't pass it on) the state is 512 bytes and the copy
//! reads 8 bytes past the stack's top. On the main thread the page there is the guard page of
//! Rust's own alternate stack: the handler faults a second time and the process dies with no dump.
//!
//! [`keep_handler_in_frame`] puts a small handler in front of crash-handler's for the signals it
//! catches: when the context would be read past the top of the stack, crash-handler gets a copy of
//! the part that is there (the rest zero: the kernel wrote nothing there). Otherwise it gets the
//! kernel's context itself, as before.
//!
//! The only `unsafe` code of this crate: system calls, and the handler itself. macOS needs none of
//! it: crash-handler takes Mach exceptions there, with no signal frame.
#![allow(unsafe_code)]

use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use crash_handler::CrashContext;

/// The signals crash-handler catches (its `EXCEPTION_SIGNALS`).
const SIGNALS: [libc::c_int; 6] = [
    libc::SIGABRT,
    libc::SIGBUS,
    libc::SIGFPE,
    libc::SIGILL,
    libc::SIGSEGV,
    libc::SIGTRAP,
];

type Handler = unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void);

/// crash-handler's own handler, which [`in_frame`] calls; 0 until [`keep_handler_in_frame`] found
/// it.
static THEIRS: AtomicUsize = AtomicUsize::new(0);

/// Put [`in_frame`] in front of crash-handler's handler; call it right after
/// `CrashHandler::attach`. True when the signals go through it.
pub fn keep_handler_in_frame() -> bool {
    let ours = in_frame as Handler as usize;
    let mut routed = false;
    for sig in SIGNALS {
        // SAFETY: a zeroed `sigaction` is a valid value for it (no handler, empty mask), and
        // sigaction only reads the signal's current action into it.
        let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
        // SAFETY: as above; the pointers are to this frame's `sa` and null.
        if unsafe { libc::sigaction(sig, std::ptr::null(), &mut sa) } != 0 {
            continue;
        }
        if sa.sa_sigaction == ours {
            routed = true;
            continue;
        }
        // crash-handler installs one handler, with SA_SIGINFO, for all six: the first one found
        // is it (SIGABRT's, just attached), and a signal left to another handler isn't touched.
        if sa.sa_flags & libc::SA_SIGINFO == 0 {
            continue;
        }
        let theirs = match THEIRS.compare_exchange(
            0,
            sa.sa_sigaction,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => sa.sa_sigaction,
            Err(found) => found,
        };
        if sa.sa_sigaction != theirs {
            continue;
        }
        // The same mask and flags (SA_ONSTACK, SA_SIGINFO), only the function is ours.
        sa.sa_sigaction = ours;
        // SAFETY: `sa` is crash-handler's own action with `in_frame`, an `extern "C"` handler of
        // the SA_SIGINFO kind, in its place; THEIRS was set above, before a signal can reach it.
        if unsafe { libc::sigaction(sig, &sa, std::ptr::null_mut()) } == 0 {
            routed = true;
        }
    }
    routed
}

/// The handler in front of crash-handler's: the kernel's context, or the part of it inside the
/// alternate stack.
unsafe extern "C" fn in_frame(sig: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let theirs = THEIRS.load(Ordering::Acquire);
    // SAFETY: THEIRS holds crash-handler's handler, a function of this type, from before this
    // handler was installed (keep_handler_in_frame).
    let theirs = unsafe { std::mem::transmute::<usize, Handler>(theirs) };
    // SAFETY: a zeroed `stack_t` is a valid value for it (no stack).
    let mut ss: libc::stack_t = unsafe { std::mem::zeroed() };
    // SAFETY: sigaltstack (a plain system call, safe in a signal handler) only reads this
    // thread's alternate stack into `ss`.
    let on_alt_stack = unsafe { libc::sigaltstack(std::ptr::null(), &mut ss) } == 0
        && ss.ss_flags & libc::SS_ONSTACK != 0;
    let stack = on_alt_stack.then(|| ss.ss_sp as usize..ss.ss_sp as usize + ss.ss_size);
    let len = context_len();
    let readable = readable(uc as usize, len, stack);
    if readable == len {
        // SAFETY: crash-handler's handler with the kernel's own arguments, as without this one.
        unsafe { theirs(sig, info, uc) };
        return;
    }
    with_copy(
        |c| &c.context,
        |copy| {
            // SAFETY: the first `readable` bytes at `uc` are the kernel's frame, inside the
            // alternate stack; `copy` is a zeroed context of the same type, as large as `len`.
            unsafe {
                std::ptr::copy_nonoverlapping(uc.cast::<u8>(), copy.cast::<u8>(), readable);
                theirs(sig, info, copy);
            }
        },
    );
}

/// The bytes crash-handler copies from a signal's context: its `CrashContext::context`.
fn context_len() -> usize {
    fn size_of_field<F>(_: fn(&CrashContext) -> &F) -> usize {
        std::mem::size_of::<F>()
    }
    size_of_field(|c| &c.context)
}

/// A zeroed context of the type crash-handler copies, on this stack, for `f`.
fn with_copy<F>(_: fn(&CrashContext) -> &F, f: impl FnOnce(*mut libc::c_void)) {
    let mut copy = MaybeUninit::<F>::zeroed();
    f(copy.as_mut_ptr().cast());
}

/// How many of the `len` bytes of a context at `at` can be read: those below the top of the
/// alternate stack (`stack`) when the frame is on it, all of them on the thread's own stack
/// (`None`: the interrupted code's frames are above it).
fn readable(at: usize, len: usize, stack: Option<Range<usize>>) -> usize {
    match stack {
        Some(s) if s.contains(&at) => len.min(s.end - at),
        _ => len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame of #1239 (a virtual machine without XSAVE): crash-handler's 16 KiB stack, the
    /// context 960 bytes under its top; 968 bytes copied would cross into the guard page above.
    #[test]
    fn a_context_is_read_up_to_the_top_of_the_alternate_stack() {
        let stack = 0x7fff_f7fb_4000..0x7fff_f7fb_8000;
        assert_eq!(readable(stack.end - 960, 968, Some(stack.clone())), 960);
        // With XSAVE the processor's state is larger: the frame starts lower, all of it is read.
        assert_eq!(readable(stack.end - 2_000, 968, Some(stack.clone())), 968);
        // On the thread's own stack, and anywhere else, as before.
        assert_eq!(readable(stack.end - 960, 968, None), 968);
        assert_eq!(readable(stack.end + 4_096, 968, Some(stack)), 968);
    }

    /// crash-handler's copy is glibc's `ucontext_t`, larger than the kernel's frame below the
    /// processor's state (304 bytes of `ucontext`, then 128 of signal info): what reads past it.
    #[test]
    fn the_copy_is_glibcs_context() {
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(context_len(), 968);
            assert_eq!(context_len(), std::mem::size_of::<libc::ucontext_t>());
        }
        assert!(context_len() > 0);
    }

    /// Attached, crash-handler's six signals go through `in_frame`, with its mask and flags; a
    /// second call changes nothing.
    #[test]
    fn the_signals_go_through_in_frame() {
        use crash_handler::{CrashEventResult, CrashHandler, make_crash_event};
        // SAFETY: the closure does nothing; no signal comes in this test.
        let event = unsafe { make_crash_event(|_| CrashEventResult::Handled(false)) };
        let handler = CrashHandler::attach(event).unwrap();
        assert!(keep_handler_in_frame());
        assert!(keep_handler_in_frame());
        let ours = in_frame as Handler as usize;
        for sig in SIGNALS {
            // SAFETY: as in keep_handler_in_frame.
            let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe { libc::sigaction(sig, std::ptr::null(), &mut sa) },
                0
            );
            assert_eq!(sa.sa_sigaction, ours, "signal {sig}");
            assert_ne!(sa.sa_flags & libc::SA_ONSTACK, 0);
            assert_ne!(sa.sa_flags & libc::SA_SIGINFO, 0);
            // SAFETY: a query of the action's mask.
            assert_eq!(unsafe { libc::sigismember(&sa.sa_mask, libc::SIGSEGV) }, 1);
        }
        assert_ne!(THEIRS.load(Ordering::Acquire), 0);
        assert_ne!(THEIRS.load(Ordering::Acquire), ours);
        drop(handler);
    }

    #[test]
    fn the_copy_starts_zeroed() {
        with_copy(
            |c| &c.context,
            |copy| {
                // SAFETY: `copy` is `context_len()` bytes, zeroed, on this stack.
                let bytes = unsafe { std::slice::from_raw_parts(copy.cast::<u8>(), context_len()) };
                assert!(bytes.iter().all(|&b| b == 0));
            },
        );
    }
}
