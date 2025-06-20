int main(void) {
    int a = 10;
    int b;
    {
        int a = 20;
        b = a;
    }
    return a;
}