int main(void) {
    int i = 0;
    int a = 0;
    do {
        i++;
        if (i % 2 == 0)
            continue;
        a += i;
    } while(i < 10);
    return a;
}