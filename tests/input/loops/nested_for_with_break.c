int main(void) {
    int i = 0;
    int j = 0;
    int a = 0;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 10; j++)
            if ((i + j) % 5 == 0)
                break;
            else a += (i + j);
    }
    return a;
}