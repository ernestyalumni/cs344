cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit. 
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

==================================================

Changes by me (Ernest Yeung)
==================================================

# Summary of changes

- *Added* (entirely *new*) Lesson 1 Code Snippets in [Lesson Code Snippets](https://github.com/ernestyalumni/cs344/tree/master/Lesson%20Code%20Snippets)


| codename          | directory                      | Keywords      | Description |
| ----------------- | :----------------------------- | :-----------: | ----------- | 
| student_func00.cu | `./Problem Sets/Problem Set 1/` | Problem Set 1 | My first attempt before I spent 2.5 months with CUDA C/C++ (about June 2016) |
| student_func.cu | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my solution |
| student_func00.cu | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my solution; has the "naive" gaussian blur method (i.e. from global memory) |
| Makefile | `./Problem Sets/Problem Set 2/` | Problem Set 2 | changed Makefile to run on my Fedora Linux setup (mostly changed gcc to nvcc compiler, needed for `cuda_runtime.h` |
| HW2 | `./Problem Sets/Problem Set 2/` | Problem Set 2 | executable for Problem Set 2 for reference (of a working executable), using the "naive" gaussian blur method (no shared memory).  Results I obtained for running `./HW2 cinque_terre_small.jpg` was `Your code ran in: 1.595616 msecs` on a NVIDIA GTX GeForce 980 Ti, EVGA, for thread block size of 16x16, for 32x32, 1.514528 msecs |


## more on Problem Set 2

My writeup is in [`CompPhys.pdf`](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf) of the [`LaTeXandpdfs` directory](https://github.com/ernestyalumni/CompPhys/tree/master/LaTeXandpdfs) of the [CompPhys github repository](https://github.com/ernestyalumni/CompPhys) - search for "On Problem Set 2".  

### on the "naive" gaussian blur method (i.e. from global memory)

#### Benchmarks:

Doing `./HW2 cinque_terre_small.jpg`, using this "naive" gaussian blur method (i.e. from global memory only), 

For a `dim3 blockSize(1,1)`, i.e. $(M_x,M_y)=(1,1)$
