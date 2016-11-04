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
- *Most definitive* implementation of **Hillis/Steele** and **Blelloch (i.e. prefix)** scan(s), coded up to CUDA C++11 standards, using global memory, in [Lesson 3 Code Snippets' `scan` subdirectory](https://github.com/ernestyalumni/cs344/tree/master/Lesson%20Code%20Snippets/Lesson%203%20Code%20Snippets/scan).   


| codename          | directory                      | Keywords      | Description |
| ----------------- | :----------------------------- | :-----------: | ----------- | 
| `student_func00.cu` | `./Problem Sets/Problem Set 1/` | Problem Set 1 | My first attempt before I spent 2.5 months with CUDA C/C++ (about June 2016) |
| `student_func.cu` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my solution; it implements shared memory for the "tiling" scheme, looping through all the regular and halo cells to load into shared memory |
| `student_func00.cu` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my solution; has the "naive" gaussian blur method (i.e. from global memory) |
| `Makefile` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | changed Makefile to run on my Fedora Linux setup (mostly changed gcc to nvcc compiler, needed for `cuda_runtime.h` |
| `HW2` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | executable for Problem Set 2 for reference (of a working executable), using the "naive" gaussian blur method (no shared memory).  Results I obtained for running `./HW2 cinque_terre_small.jpg` was `Your code ran in: 1.595616 msecs` on a NVIDIA GTX GeForce 980 Ti, EVGA, for thread block size of 16x16, for 32x32, 1.514528 msecs; see the benchmarks below |
| `student_func_global.cu` | `./Problem Sets/Problem Set 2/` | Problem Set 2 | my final version implementing the "naive" gaussian blur method (i.e. from global memory) |
| `reduce_serial.c` | `./Lesson Code Snippets/Lesson 3 Code Snippets/`  | Lesson 3 Code Snippet, reduce, serial, C | [Serial implementation of reduce](https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876789040923#), in C   |
| `reduce_serial_vectors.cpp` | `./Lesson Code Snippets/Lesson 3 Code Snippets/`  | Lesson 3 Code Snippet, reduce, serial, C++11/14, vector, vectors |  [Serial implementation of reduce](https://classroom.udacity.com/courses/cs344/lessons/86719951/concepts/876789040923#), using C++11/14 vector(s) (library); next step I'd take is to write functions, but using templates   |
| `bitwiserightshift.cpp` | `./Lesson Code Snippets/Lesson 3 Code Snippets/` | Lesson 3 Code Snippet, bitwise right shift, bitwise operator | explanation, exploration of bitwise right shift, bitwise operators |


## more on Problem Set 2

My writeup is in [`CompPhys.pdf`](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf) of the [`LaTeXandpdfs` directory](https://github.com/ernestyalumni/CompPhys/tree/master/LaTeXandpdfs) of the [CompPhys github repository](https://github.com/ernestyalumni/CompPhys) - search for "On Problem Set 2".  

### on the "naive" gaussian blur method (i.e. from global memory)

This method was implemented in [`student_func_global.cu`](https://github.com/ernestyalumni/cs344/blob/master/Problem%20Sets/Problem%20Set%202/student_func_global.cu)

#### Benchmarks (global memory):

Doing `./HW2 cinque_terre_small.jpg`, using this "naive" gaussian blur method (i.e. from global memory only), on the *EVGA NVIDIA GeForce GTX 980 Ti*:

For a `dim3 blockSize(1,1)`, i.e. $(M_x,M_y)=(1,1)$

```  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg  
Your code ran in: 46.665215 msecs.  
PASS  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg  
Your code ran in: 46.587330 msecs.  
PASS  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg  
Your code ran in: 46.565727 msecs.  
PASS  
```

For a `dim3 blockSize(2,2)`

```  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 12.110336 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 12.115456 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 12.149376 msecs.
PASS
```

For a `dim3 blockSize(4,4)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 3.383424 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 3.057344 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 3.060672 msecs.
PASS
```

For a `dim3 blockSize(8,8)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.775840 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.781280 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.776864 msecs.
PASS
```

For a `dim3 blockSize(16,16)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.582560 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.569120 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.350688 msecs.
PASS
```

For a `dim3 blockSize(32,32)`

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.521856 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.530208 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.513664 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.514176 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.305984 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.297568 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.303168 msecs.
PASS
```

For block sizes greater than 32, we obtain an error.  So it appears that for this "naive" gaussian blur (i.e. only global memory), then setting `dim3 blockSize(16,16)` or `dim3 blockSize(32,32)` makes the code run the fastest.

### Examples of using `__shared__` memory

I provide further, real-world, in practice, examples of using `__shared__` (more examples, the better, as otherwise we only have the 1 example from the documentation):  

- From [eliben's fork of cs344](https://github.com/eliben/cs344/blob/master/HW2/student_func.cu):

```
__global__ void blur_shared( ... ) {

	   extern __shared__ float sfilter[];

}

```

- From [`heat_2d.cu`](https://github.com/ernestyalumni/CUDACFD_out/blob/master/heat2d/physlib/heat_2d.cu):

```
__global__ void tempKernel( ... ) {
	   
	   extern __shared__ float s_in[];

}
```

### On other people's implementation/solutions for Problem Set 2 (Udacity CS344) and the implementation of `__shared__` memory tiling scheme

The "naive" global memory implementation of image blurring (really, the use of a "local" stencil) is fairly clear and straightforward as it really is 1-to-1 from global memory to global memory.  A great majority of code solutions/implementations floating out there on github and forums are of this.  See my [`student_func_global.cu`](https://github.com/ernestyalumni/cs344/blob/master/Problem%20Sets/Problem%20Set%202/student_func_global.cu).

No one seems to have a clear, lucid explanation or grasp of the "tiling" scheme needed to implement `__shared__` memory, and in particular, showing the "short strokes" that would account for, comprehensively and **correctly** the, literal, corner cases.

For instance, [raoqiyu's solution for Problem Set 2](https://github.com/raoqiyu/CS344-Problem-Sets/tree/master/Problem%20Set%202), in particular, [raoqiyu's `student_func.cu`](https://github.com/raoqiyu/CS344-Problem-Sets/blob/master/Problem%20Set%202/student_func.cu), fails to account for the corner cases:

```  
Your code ran in: 1.399648 msecs.
Difference at pos 0
Reference: 255
GPU      : 214
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.388672 msecs.
Difference at pos 0
Reference: 255
GPU      : 214
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.401088 msecs.
Difference at pos 0
Reference: 255
GPU      : 214  
```

Same as in the case of [tmoneyx01's solution for Problem Set 2 i.e. Homework 2](https://github.com/tmoneyx01/Udacity_CS344/tree/master/HW2), in particular, [tmoneyx01's `student_func.cu`](https://github.com/tmoneyx01/Udacity_CS344/blob/master/HW2/student_func.cu)


```  
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.387744 msecs.
Difference at pos 0
Reference: 255
GPU      : 214
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.380704 msecs.
Difference at pos 0
Reference: 255
GPU      : 214
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.384992 msecs.
Difference at pos 0
Reference: 255
GPU      : 214
```  

In both cases, the problem seems to arise from not accounting for the "corner cases" of the so-called "halo cells" and this is clearly seen by checking out the image outputted `HW2_differenceImage.png`.  

`ruizhou_313809205764` or [ruizhou](https://discussions.udacity.com/users/ruizhou_313809205764/activity) on the Udacity cs344 forums had an interesting scheme where the shared tile was inputted in, in 4 steps, i.e. in 4 overlapping tiles, with the overlap being the "regular" cells within a threadblock, and so all corner cases for the halo cells were included, even though the regular cells were counted in 4 times (total).  I tried to write up the code and placed it in [`student_func_sharedoverlaps.cu`](https://github.com/ernestyalumni/cs344/blob/master/Problem%20Sets/Problem%20Set%202/student_func_sharedoverlaps.cu) to test it out.  It's essentially the same, but I'm just trying to follow my mathematical notation in [`CompPhys.pdf`](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf).

However, I obtain
```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
CUDA error at: student_func.cu:468
an illegal memory access was encountered cudaGetLastError()
```
and it wasn't the problem with choice of block size (2,4,8,16).


### tiling scheme with `__shared__` memory is a nontrivial problem

The so-called "tiling" scheme with `__shared__` memory is a nontrivial problem.  In fact, there is active research in optimizing the tiling scheme or even simply clarifying the implementation of the method on the GPU.

Siham Tabik, Maurice Peemen, Nicolas Guil, and Henk Corporaal. *Demystifying the 16 x 16 thread-block for stencils on the GPU.* CONCURRENCY AND COMPUTATION: PRACTICE AND EXPERIENCE  *Concurrency Computat.: Pract. Exper.* (2015)  [CPE.pdf](http://www.ac.uma.es/~siham/CPE.pdf)

### **Solution to Problem Set 2 that works (passes) and uses `__shared__` memory**

I wrote up my best and fastest implementation, that works (passes), of gaussian blur, which uses a stencil pattern, that uses `__shared__` memory and placed it here:

[`student_func.cu`](https://github.com/ernestyalumni/cs344/blob/master/Problem%20Sets/Problem%20Set%202/student_func.cu)

The loop through the halo cells I found was *highly* non-trivial, and was not well-tackled, or handled, with if-then clauses/cases.  The loop through the regular and halo cells to be loaded was from [Samuel Lin or Samuel271828](https://discussions.udacity.com/users/Samuel271828), where his code was also placed in [Samuel Lin or samuellin3310's github repositories](https://github.com/samuellin3310), namely [`student_fuction_improved_share.cu` (sic)](https://github.com/samuellin3310/ro-to-Parallel-Programming_set2/blob/master/student_fuction_improved_share.cu).  And again, take a look at my writeup, named [`CompPhys.pdf`](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf), for the mathematical formulation.

Here the benchmarks for [`student_func.cu`](https://github.com/ernestyalumni/cs344/blob/master/Problem%20Sets/Problem%20Set%202/student_func.cu):

```
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.428704 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.420000 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.417344 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.421216 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.415008 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.181792 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.187072 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.200896 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.179328 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.181440 msecs.
PASS
[@localhost Problem Set 2]$ ./HW2 cinque_terre_small.jpg
Your code ran in: 1.184480 msecs.
PASS
```  

Curiously, the code runs at either 1.42 msecs. or 1.18 msecs.  Taking 1.18 msecs, using `__shared__` memory is an improvement over global memory of (1.30 - 1.18)/1.30 * 100 % = 9 %, 9 or 10 percent improvement.  

(EY: 20161002 The following point was resolved in the [forum discussion in Udacity cs344](https://discussions.udacity.com/t/any-one-completed-problem-set-2-using-shared-memory/158442/7), thanks again to [Samuel Lin or Samuel271828](https://discussions.udacity.com/users/Samuel271828))

One point I still don't understand is how the placement of the line
```
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
```
or, in my notation

```
if ( k_x >= numCols || k_y >= numRows ) {
   return; }
```

could affect the "correctness" of the blur function at the very edges.  When I placed it at the beginning, instead of in the middle, after loading the values into shared memory, it gave a wrong answer.  I don't see why.

Otherwise, the loop for this code through the cells is very clear in accounting for all the halo cells as well, and "corner cases" of the desired stencil.  Also, I thought this problem set was highly non-trivial with the tiling scheme for shared memory, as there are a lot of incorrect code out that fails to implement this correctly.  

(EY: 20161002 Again, this point was resolved in the [forum discussion in Udacity cs344](https://discussions.udacity.com/t/any-one-completed-problem-set-2-using-shared-memory/158442/7), thanks again to [Samuel Lin or Samuel271828](https://discussions.udacity.com/users/Samuel271828) - also see [`CompPhys.pdf`, search for Problem Set 2](https://github.com/ernestyalumni/CompPhys/blob/master/LaTeXandpdfs/CompPhys.pdf)).  

