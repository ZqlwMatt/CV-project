import java.util.Scanner;  
public class AplusB { 
    private static Scanner sc;
    public static void main(String[] args) {
        sc = new Scanner(System.in); 
        int a = sc.nextInt(), b = sc.nextInt();
        System.out.println(a + b);
    }
}