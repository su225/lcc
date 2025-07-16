int main(void) {
    int a = 0;
    int i = 0;
    for (;;) {
        a += i;
        i++;
        if (i >= 10)
            break;
    }
    return a;
}