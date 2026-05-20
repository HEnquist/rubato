//! 32-byte aligned heap buffer for sinc tables and the combined-sinc scratch.
//!
//! A plain `Vec<f64>` only carries the type's natural alignment (8 bytes), which
//! the system allocator rounds up to 16. That's enough for SSE/NEON 128-bit loads
//! but not for AVX 256-bit loads: when the base lands at offset 16 mod 32, half
//! of the loads cross a 64-byte cache line boundary and incur a split penalty
//! (~5 cycles on Intel, ~12 on AMD). For an f64 sinc that's 32 splits per dot
//! product, which more than cancels the FMA-throughput gains from going to four
//! accumulator chains.

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

const ALIGN: usize = 32;

/// Heap buffer of `T` with 32-byte alignment.
///
/// Only safe to call `zeroed` for types whose all-zero bit pattern is a valid
/// value (true for the numeric `Sample` types we use here).
#[cfg_attr(feature = "bench_asyncro", visibility::make(pub))]
pub(crate) struct AlignedBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for AlignedBuf<T> {}
unsafe impl<T: Sync> Sync for AlignedBuf<T> {}

impl<T: Copy> AlignedBuf<T> {
    fn layout(len: usize) -> Layout {
        let size = len
            .checked_mul(mem::size_of::<T>())
            .expect("AlignedBuf size overflow");
        Layout::from_size_align(size, ALIGN).expect("invalid AlignedBuf layout")
    }

    pub(crate) fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                _marker: PhantomData,
            };
        }
        let layout = Self::layout(len);
        let raw = unsafe { alloc::alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    pub(crate) fn from_slice(src: &[T]) -> Self {
        if src.is_empty() {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                _marker: PhantomData,
            };
        }
        let layout = Self::layout(src.len());
        let raw = unsafe { alloc::alloc(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.as_ptr(), src.len());
        }
        Self {
            ptr,
            len: src.len(),
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            let size = self.len * mem::size_of::<T>();
            let layout = Layout::from_size_align(size, ALIGN).unwrap();
            unsafe { alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

impl<T> Deref for AlignedBuf<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedBuf<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
