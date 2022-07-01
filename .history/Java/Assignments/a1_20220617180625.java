import java.util.*;
public class a1 {
    private static Scanner sc;
    public static int getnum(String str) {
        int res = 0;
        return res;
    }
    public static void main(String[] args) {
        sc = new Scanner(System.in);
        String _str = sc.nextLine();

        String ans = "";
        char str[] = _str.toCharArray();
        
        int pos = 0, cnt = 0;
        int cof[] = new int[100];
        int order[] = new int[100];

        for(int i = 0; i < _str.length(); ++i) {
            if(str[i] == '*') {
                cof[cnt] = getnum(_str.substring(pos, i));
            }
            else if(i!=1 && (str[i] == '+' || str[i] == '-')) {

            }
        }
    }
}
