# bookish-octo-engine
Just a simple cli program in C to test libnoodlenet https://github.com/gnaservicesinc/libnoodlenet

Accepts 2 arguments. The first is a path to a model exported with libnoodlenet and the second is a path to an image. It will output the result, true (the image contains the object), false, it does not, or throw an error.

## Building and Installation

### Prerequisites
- C compiler (gcc or clang)
- Autotools (autoconf, automake)
- libnoodlenet library and headers

### Build Instructions
```bash
# Generate the configure script
autoreconf --install

# Configure the build
./configure

# Build the program
make

# Install the program (default: /usr/local/bin)
sudo make install
```

### Usage
```bash
bookish-octo-engine <model_path> <image_path>
```

