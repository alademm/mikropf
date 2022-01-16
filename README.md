# MikroPF
A lightweight and multi-threaded particle filter library in C++.

# Usage
MikroPF is a single class contained in `MikroPF.h` and `MikroPF.cpp`. It has no external dependency and no specific build process is required. You can just add it to your project and start using it.

Note that dynamic memory allocation and thread creation happen during initialization only. This makes the update loop more efficient for online use.
