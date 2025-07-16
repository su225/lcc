int main(void) {
    int a = 0;
    int i = 0;
    while (i < 10) {
        i++;
        if (i % 2 == 0)
            continue;
        a += i;
    }
    return a;
}