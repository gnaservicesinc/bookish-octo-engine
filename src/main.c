// Simple CLI for libnoodlenet
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <noodlenet.h>

typedef struct {
    NN_Model* model;
    int model_locked; // locked after load or first training
    int dirty_since_save;
    // Training config
    int steps;
    int batch_size;
    float learning_rate;
    float l1_lambda;
    float l2_lambda;
    int shuffle; // 0/1
    char pos_dir[1024];
    char neg_dir[1024];
    char val_dir[1024];
    // Visualization options
    int vis_mode;     // 0=weights, 1=heatmap
    int vis_scale;    // 0=minmax, 1=symmetric-zero
    int vis_bias;     // 0/1 include biases
    int vis_stats;    // 0/1 include stats
    int vis_raw;      // 0/1 raw non-square weights
} AppState;

static void clear_state(AppState* s) {
    if (s->model) { nn_model_free(s->model); s->model = NULL; }
    s->model_locked = 0; s->dirty_since_save = 0;
    s->steps = 100; s->batch_size = 10; s->learning_rate = 0.01f; s->l1_lambda = 0.0f; s->l2_lambda = 0.0f; s->shuffle = 1;
    s->pos_dir[0] = s->neg_dir[0] = s->val_dir[0] = '\0';
    s->vis_mode = 0; s->vis_scale = 0; s->vis_bias = 0; s->vis_stats = 0; s->vis_raw = 0;
}

static void pause_enter() {
    printf("\nPress ENTER to continue...");
    int c; while ((c = getchar()) != '\n' && c != EOF) {}
}

static void read_line(const char* prompt, char* buf, size_t n) {
    printf("%s", prompt);
    fflush(stdout);
    if (fgets(buf, (int)n, stdin)) {
        size_t len = strlen(buf);
        if (len && buf[len-1] == '\n') buf[len-1] = '\0';
        // Trim trailing spaces/tabs/carriage returns
        len = strlen(buf);
        while (len > 0 && (buf[len-1] == ' ' || buf[len-1] == '\t' || buf[len-1] == '\r')) { buf[len-1] = '\0'; len--; }
    } else {
        buf[0] = '\0'; clearerr(stdin);
    }
}

static int is_dir(const char* path) {
    struct stat st; if (stat(path, &st) != 0) return 0; return S_ISDIR(st.st_mode);
}

static void print_menu() {
    printf("\n=== NoodleNet CLI ===\n");
    printf("1) Define new model\n");
    printf("2) Define training task\n");
    printf("3) Run training\n");
    printf("4) Save model\n");
    printf("5) Load model\n");
    printf("6) Inspect model\n");
    printf("7) Test model (file or directory)\n");
    printf("8) Export layer visualizations\n");
    printf("9) Set optimizer (SGD/RMSprop/Adam)\n");
    printf("10) Configure visualization options\n");
    printf("11) Exit\n");
}

static ActivationFunction ask_activation(const char* prompt) {
    char buf[64];
    printf("%s [sigmoid|tanh|relu|leaky_relu]: ", prompt);
    if (!fgets(buf, sizeof(buf), stdin)) return NN_ACTIVATION_FUNCTION_SIGMOID;
    if (strstr(buf, "tanh")) return NN_ACTIVATION_FUNCTION_TANH;
    if (strstr(buf, "relu") && strstr(buf, "leaky")) return NN_ACTIVATION_FUNCTION_LEAKY_RELU;
    if (strstr(buf, "relu")) return NN_ACTIVATION_FUNCTION_RELU;
    return NN_ACTIVATION_FUNCTION_SIGMOID;
}

static void action_set_optimizer(AppState* s) {
    if (!s->model) { printf("Define or load a model first.\n"); return; }
    printf("Select optimizer: 0) SGD, 1) RMSprop, 2) Adam: ");
    char line[64]; if (!fgets(line, sizeof(line), stdin)) return; int opt = atoi(line);
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    if (opt == 1) { // RMSprop
        // Optionally ask for beta2/epsilon
        printf("RMSprop beta2 (default 0.999): "); if (fgets(line, sizeof(line), stdin)) { float v=(float)atof(line); if (v>0&&v<1) beta2=v; }
        printf("epsilon (default 1e-8): "); if (fgets(line, sizeof(line), stdin)) { float v=(float)atof(line); if (v>0) eps=v; }
    } else if (opt == 2) { // Adam
        printf("Adam beta1 (default 0.9): "); if (fgets(line, sizeof(line), stdin)) { float v=(float)atof(line); if (v>0&&v<1) beta1=v; }
        printf("Adam beta2 (default 0.999): "); if (fgets(line, sizeof(line), stdin)) { float v=(float)atof(line); if (v>0&&v<1) beta2=v; }
        printf("epsilon (default 1e-8): "); if (fgets(line, sizeof(line), stdin)) { float v=(float)atof(line); if (v>0) eps=v; }
    }
    NN_Optimizer o = NN_OPTIMIZER_SGD;
    if (opt == 1) o = NN_OPTIMIZER_RMSPROP; else if (opt == 2) o = NN_OPTIMIZER_ADAM;
    if (nn_model_set_optimizer(s->model, o, beta1, beta2, eps) == 0) printf("Optimizer updated.\n"); else printf("Failed to set optimizer.\n");
}

static void action_define_model(AppState* s) {
    if (s->model_locked) { printf("Model is locked (loaded or trained). Start a new process to define a new model.\n"); return; }
    if (s->model) { nn_model_free(s->model); s->model = NULL; }
    int input = 512*512; int output = 1;
    char line[128];
    printf("Input neurons fixed: %d\n", input);
    printf("Output neurons fixed: %d\n", output);
    printf("Number of hidden layers: ");
    if (!fgets(line, sizeof(line), stdin)) return; int num_hidden = atoi(line);
    if (num_hidden < 0) num_hidden = 0; if (num_hidden > 8) num_hidden = 8;
    int* sizes = (int*)malloc(sizeof(int)*num_hidden);
    ActivationFunction* acts = (ActivationFunction*)malloc(sizeof(ActivationFunction)*num_hidden);
    for (int i = 0; i < num_hidden; ++i) {
        printf("Hidden layer %d size: ", i+1);
        if (!fgets(line, sizeof(line), stdin)) { sizes[i]=128; } else sizes[i] = atoi(line);
        if (sizes[i] <= 0) sizes[i] = 128;
        acts[i] = ask_activation("Activation");
    }
    ActivationFunction out_act = ask_activation("Output activation");
    s->model = nn_model_create(input, sizes, num_hidden, output, acts, out_act);
    free(sizes); free(acts);
    if (s->model) { printf("Model created.\n"); s->dirty_since_save = 1; } else printf("Failed to create model.\n");
}

static void action_define_training(AppState* s) {
    char line[2048];
    printf("Steps (epochs/updates): "); if (fgets(line, sizeof(line), stdin)) s->steps = atoi(line);
    printf("Batch size: "); if (fgets(line, sizeof(line), stdin)) s->batch_size = atoi(line);
    printf("Learning rate: "); if (fgets(line, sizeof(line), stdin)) s->learning_rate = (float)atof(line);
    printf("L1 lambda (0 for off): "); if (fgets(line, sizeof(line), stdin)) s->l1_lambda = (float)atof(line);
    printf("L2 lambda (0 for off): "); if (fgets(line, sizeof(line), stdin)) s->l2_lambda = (float)atof(line);
    printf("Shuffle each epoch? [1=yes,0=no]: "); if (fgets(line, sizeof(line), stdin)) { int v = atoi(line); s->shuffle = (v!=0); }
    read_line("Required: Positive images dir: ", s->pos_dir, sizeof(s->pos_dir));
    if (!is_dir(s->pos_dir)) printf("Warning: directory not found.\n");
    read_line("Optional: Negative images dir (empty to skip): ", s->neg_dir, sizeof(s->neg_dir));
    if (s->neg_dir[0] && !is_dir(s->neg_dir)) printf("Warning: directory not found.\n");
    read_line("Optional: Validation images dir (empty to skip): ", s->val_dir, sizeof(s->val_dir));
    if (s->val_dir[0] && !is_dir(s->val_dir)) printf("Warning: directory not found.\n");
}

static void action_run_training(AppState* s) {
    if (!s->model) { printf("Define or load a model first.\n"); return; }
    if (!s->pos_dir[0]) { printf("Define training task with positive images first.\n"); return; }
    float last_loss = 0.0f, val_loss = 0.0f;
    int rc = nn_train_from_dirs(s->model, s->pos_dir, s->neg_dir[0]?s->neg_dir:NULL, s->val_dir[0]?s->val_dir:NULL,
                                s->steps, s->batch_size, s->shuffle, s->learning_rate, s->l1_lambda, s->l2_lambda, &last_loss, &val_loss);
    if (rc == 0) {
        if (s->val_dir[0]) printf("Training complete. Last loss: %.6f, Val loss: %.6f\n", last_loss, val_loss);
        else printf("Training complete. Last loss: %.6f\n", last_loss);
        s->model_locked = 1;
        s->dirty_since_save = 1;
    }
    else printf("Training failed.\n");
}

static void action_save_model(AppState* s) {
    if (!s->model) { printf("No model to save.\n"); return; }
    char path[1024]; read_line("Save to path (.senm): ", path, sizeof(path));
    // Persist metadata
    (void)nn_model_set_data_dirs(s->model, s->pos_dir[0]?s->pos_dir:NULL, s->neg_dir[0]?s->neg_dir:NULL, s->val_dir[0]?s->val_dir:NULL);
    (void)nn_model_set_locked_training_params(s->model, s->batch_size, s->learning_rate, s->shuffle, NN_OPTIMIZER_SGD);
    int rc = nn_model_save(s->model, path);
    if (rc == 0) { printf("Saved to %s\n", path); s->dirty_since_save = 0; } else printf("Failed to save.\n");
}

static void action_load_model(AppState* s) {
    char path[1024]; read_line("Load from path (.senm): ", path, sizeof(path));
    NN_Model* m = nn_model_load(path);
    if (!m) { printf("Failed to load model.\n"); return; }
    if (s->model) nn_model_free(s->model);
    s->model = m; s->model_locked = 1; s->dirty_since_save = 0; printf("Model loaded.\n");
    // Load stored data dirs if present
    const char *p=NULL, *n=NULL, *v=NULL;
    if (nn_model_get_data_dirs(s->model, &p, &n, &v) == 0) {
        if (p) strncpy(s->pos_dir, p, sizeof(s->pos_dir)-1);
        if (n) strncpy(s->neg_dir, n, sizeof(s->neg_dir)-1);
        if (v) strncpy(s->val_dir, v, sizeof(s->val_dir)-1);
    }
}

static const char* act_name(ActivationFunction a) {
    switch (a) {
        case NN_ACTIVATION_FUNCTION_TANH: return "tanh";
        case NN_ACTIVATION_FUNCTION_RELU: return "relu";
        case NN_ACTIVATION_FUNCTION_LEAKY_RELU: return "leaky_relu";
        case NN_ACTIVATION_FUNCTION_SIGMOID: default: return "sigmoid";
    }
}

static void action_inspect_model(const AppState* s) {
    if (!s->model) { printf("No model defined/loaded.\n"); return; }
    int H = nn_model_num_hidden(s->model);
    printf("Input: %d, Hidden layers: %d, Output: %d\n", nn_model_input_size(s->model), H, nn_model_output_size(s->model));
    for (int i=0;i<H;++i) {
        printf("  Hidden %d: size=%d, act=%s\n", i+1, nn_model_hidden_size(s->model,i), act_name(nn_model_hidden_activation(s->model,i)));
    }
    printf("  Output act: %s\n", act_name(nn_model_output_activation(s->model)));
    const char *p=NULL,*n=NULL,*v=NULL; if (nn_model_get_data_dirs(s->model,&p,&n,&v)==0) {
        printf("Data dirs: pos=%s, neg=%s, val=%s\n", p?p:"(none)", n?n:"(none)", v?v:"(none)");
    }
}

static int list_images(const char* dir, char*** out, int* n) {
    *out=NULL; *n=0; DIR* d=opendir(dir); if(!d) return -1; struct dirent* e; int cap=32; char** arr=malloc(sizeof(char*)*cap); if(!arr){closedir(d);return -1;}
    while((e=readdir(d))){ if(e->d_name[0]=='.') continue; char p[2048]; snprintf(p,sizeof(p),"%s/%s",dir,e->d_name); struct stat st; if(stat(p,&st)!=0||!S_ISREG(st.st_mode)) continue; if(!strstr(p, ".png")&&!strstr(p, ".bmp")&&!strstr(p, ".jpg")&&!strstr(p, ".jpeg")) continue; if(*n>=cap){cap*=2; char** t=realloc(arr,sizeof(char*)*cap); if(!t){for(int i=0;i<*n;++i)free(arr[i]);free(arr);closedir(d);return -1;} arr=t;} arr[*n]=strdup(p); if(!arr[*n]){for(int i=0;i<*n;++i)free(arr[i]);free(arr);closedir(d);return -1;} (*n)++; }
    closedir(d); *out=arr; return 0; }

static void free_list(char** a, int n){ if(!a) return; for(int i=0;i<n;++i) free(a[i]); free(a);} 

static void action_test_model(const AppState* s) {
    if (!s->model) { printf("No model defined/loaded.\n"); return; }
    char path[1024]; read_line("Image file or directory: ", path, sizeof(path));
    struct stat st; if (stat(path, &st)!=0) { printf("Not found.\n"); return; }
    if (S_ISDIR(st.st_mode)) {
        char** imgs=NULL; int n=0; if (list_images(path,&imgs,&n)!=0||n==0){ printf("No images found.\n"); return; }
        int tp=0; for(int i=0;i<n;++i){ float p=0.0f; if (nn_model_predict_image(s->model, imgs[i], &p)==0){ printf("%s => %.6f (%s)\n", imgs[i], p, p>0.5f?"true":"false"); if(p>0.5f) tp++; } }
        printf("Total: %d, Positives: %d\n", n, tp);
        free_list(imgs,n);
    } else {
        float p=0.0f; if (nn_model_predict_image(s->model, path, &p)==0) printf("%s => %.6f (%s)\n", path, p, p>0.5f?"true":"false"); else printf("Prediction failed.\n");
    }
}

static void action_export_visuals(const AppState* s) {
    if (!s->model) { printf("No model defined/loaded.\n"); return; }
    char out[1024]; read_line("Output directory: ", out, sizeof(out));
    NN_VisOptions opts; opts.mode = (s->vis_mode==0)?NN_VIS_MODE_WEIGHTS:NN_VIS_MODE_HEATMAP; opts.scale = (s->vis_scale==0)?NN_VIS_SCALE_MINMAX:NN_VIS_SCALE_SYM_ZERO; opts.include_bias = s->vis_bias; opts.include_stats = s->vis_stats; opts.raw_weights_full = s->vis_raw;
    int rc = nn_export_layer_visualizations_ex(s->model, out, &opts);
    if (rc==0) printf("Exports written to %s (PGM files). mode=%s, scale=%s, biases=%s, stats=%s, raw=%s\n", out,
                     s->vis_mode==0?"weights":"heatmap", s->vis_scale==0?"minmax":"symzero",
                     s->vis_bias?"on":"off", s->vis_stats?"on":"off", s->vis_raw?"on":"off");
    else printf("Export failed.\n");
}

static void action_configure_vis(AppState* s) {
    printf("Visualization mode (0=weights,1=heatmap) [%d]: ", s->vis_mode); char line[64]; if (fgets(line,sizeof(line),stdin)) { int v=atoi(line); if (v==0||v==1) s->vis_mode=v; }
    printf("Scaling (0=minmax,1=symmetric zero-centered) [%d]: ", s->vis_scale); if (fgets(line,sizeof(line),stdin)) { int v=atoi(line); if (v==0||v==1) s->vis_scale=v; }
    printf("Include biases (0/1) [%d]: ", s->vis_bias); if (fgets(line,sizeof(line),stdin)) { int v=atoi(line); if (v==0||v==1) s->vis_bias=v; }
    printf("Include stats (0/1) [%d]: ", s->vis_stats); if (fgets(line,sizeof(line),stdin)) { int v=atoi(line); if (v==0||v==1) s->vis_stats=v; }
    printf("Raw weights non-square (0/1) [%d]: ", s->vis_raw); if (fgets(line,sizeof(line),stdin)) { int v=atoi(line); if (v==0||v==1) s->vis_raw=v; }
    printf("Updated: mode=%s, scale=%s, biases=%s, stats=%s, raw=%s\n",
           s->vis_mode==0?"weights":"heatmap", s->vis_scale==0?"minmax":"symzero",
           s->vis_bias?"on":"off", s->vis_stats?"on":"off", s->vis_raw?"on":"off");
}

int main(void) {
    AppState st; memset(&st, 0, sizeof(st)); clear_state(&st);
    for (;;) {
        print_menu();
        printf("Select: ");
        char line[64]; if (!fgets(line, sizeof(line), stdin)) break; int choice = atoi(line);
        switch (choice) {
            case 1: action_define_model(&st); break;
            case 2: action_define_training(&st); break;
            case 3: action_run_training(&st); break;
            case 4: action_save_model(&st); break;
            case 5: action_load_model(&st); break;
            case 6: action_inspect_model(&st); break;
            case 7: action_test_model(&st); break;
            case 8: action_export_visuals(&st); break;
            case 9: action_set_optimizer(&st); break;
            case 10: action_configure_vis(&st); break;
            case 11: clear_state(&st); printf("Goodbye.\n"); return 0;
            default: printf("Unknown option.\n"); break;
        }
        pause_enter();
    }
    clear_state(&st);
    return 0;
}
