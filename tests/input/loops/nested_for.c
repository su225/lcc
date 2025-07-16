int main(void) {
    int i = 0;
    int a = 0;
    for (i = 0; i < 10; i++)
        for (int j = 0; j < i; j++)
            a += (i+j);
    return a;
}