import java.io.*;
import java.util.*;
public class a1 {
    private static Scanner sc;
    public static int getnum(String str) {
        if(str.length() == 0) return 1;
        else return Integer.parseInt(str);
    }
    public static void main(String[] args) {
        sc = new Scanner(System.in);
        String _str = sc.nextLine();
        _str = _str.replace(" ", "");
        _str = _str.replace("^", "");
        _str = _str.replace("*", "");
        _str = _str + "+";
        
 
        String ans = "";
        char str[] = _str.toCharArray();
        
        System.out.println(_str);
        
        int pos = 0, cnt = 0, ok = 0;
        int cof[] = new int[100];
        int order[] = new int[100];

        for(int i = 0; i < _str.length(); ++i) {
            if(str[i] == 'x') {
                cof[++cnt] = getnum(_str.substring(pos, i));
                pos = i;
            }
            else if(ok == 1 && (str[i] == '+' || str[i] == '-')) {
                order[cnt] = getnum(_str.substring(pos, i));
                pos = i;
                ok = 0;
            }
        }

        System.out.println(cnt);
        
        for(int i = 1; i <= cnt; ++i) cof[i] *= order[i];
        for(int i = 1; i <= cnt; ++i) {
            if(order[i] == 0) 
            if(i > 1 && cof[i] > 0) System.out.print('+');
            System.out.print(cof[i]);
            if(order[i] != 1) System.out.print('x');
            if(order[i] != 2) {
                System.out.print("^" + Integer.toString(order[i]-1));
            }
        }
    }
}
