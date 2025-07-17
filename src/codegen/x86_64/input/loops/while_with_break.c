int main(void) {
    int a = 0;
    int i = 0;
    while (1) {
        a += i;
        i++;
        if (i >= 10)
            break;
    }
    return a;
}