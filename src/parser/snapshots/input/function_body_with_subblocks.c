int main(void) {
    int a = 10;
    int b; {
        b = 20;
        a = a + b;
    }
    return a;
}