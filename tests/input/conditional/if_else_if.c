int main(void) {
    int a = 10;
    int b;
    if (a < 5) {
        b = 1;
    } else if (a < 10) {
        b = 2;
    } else if (a < 20) {
        b = 3;
    } else {
        b = 4;
    }
    return b;
}