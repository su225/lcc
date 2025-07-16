int main(void) {
    int i = 0;
    int a = 0;
    for (i = 0; i < 10; i++) {
        if (i % 2 == 0) continue;
        a += i;
    }
    return a;
}