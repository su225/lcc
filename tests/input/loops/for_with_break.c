int main(void) {
    int a = 0;
    int i;
    for (i = 0; i < 10; i++)
        if (i % 2 != 0)
            break;
        else a++;
    return a;
}