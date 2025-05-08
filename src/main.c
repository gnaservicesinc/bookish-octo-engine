#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <noodlenet.h>

void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s <model_path> <image_path>\n", program_name);
    fprintf(stderr, "  <model_path>: Path to a model exported with libnoodlenet\n");
    fprintf(stderr, "  <image_path>: Path to an image to analyze\n");
}

int main(int argc, char *argv[]) {
    // Check if we have the correct number of arguments
    if (argc != 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    // Check if files exist
    FILE *model_file = fopen(model_path, "r");
    if (!model_file) {
        fprintf(stderr, "Error: Could not open model file '%s'\n", model_path);
        return EXIT_FAILURE;
    }
    fclose(model_file);

    FILE *image_file = fopen(image_path, "r");
    if (!image_file) {
        fprintf(stderr, "Error: Could not open image file '%s'\n", image_path);
        return EXIT_FAILURE;
    }
    fclose(image_file);

    // Process the image using noodlenet_predict
    int result = noodlenet_predict(model_path, image_path);

    if (result < 0) {
        fprintf(stderr, "Error: Failed to process image '%s'\n", image_path);
        return EXIT_FAILURE;
    }

    // Output the result
    printf("%s\n", result ? "true" : "false");

    return EXIT_SUCCESS;
}
