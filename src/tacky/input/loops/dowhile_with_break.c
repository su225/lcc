int main(void) {
    int i = 0;
    int a = 0;
    do {
        a += i;
        i++;
        if (i >= 10)
            break;
    } while(1);
    return a;
}