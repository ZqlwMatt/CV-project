import java.io.*;
import java.util.*;
public class a1 {
    private static Scanner sc;
    public static int getnum(String str) {
        
        return str.length()==0 ? 1 : Integer.parseInt(str);
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
        
        int pos = 0, cnt = 0, status = 0;
        int cof[] = new int[100];
        int order[] = new int[100];

        for(int i = 0; i < _str.length(); ++i) {
            if(str[i] == 'x') {
                cof[++cnt] = getnum(_str.substring(pos, i));
                pos = i+1;
                status = 1;
            }
            else if(str[i] == '+' || str[i] == '-') {
                if(status == 1) {
                    order[cnt] = getnum(_str.substring(pos, i));
                    status = 0;
                }
                pos = i;
            }
        }
        
        for(int i = 1; i <= cnt; ++i) cof[i] *= order[i];
        for(int i = 1; i <= cnt; ++i) {
            if(order[i] == 0) continue;
            if(i > 1 && cof[i] > 0) System.out.print('+');

            System.out.print(cof[i]);

            if(order[i] == 1) continue;
            else if(order[i] == 2) {
                if(Math.abs(cof[i]) != 1) System.out.print('*');
                System.out.print('x');
            }
            else {
                if(Math.abs(cof[i]) != 1) System.out.print('*');
                System.out.print("x^" + Integer.toString(order[i]-1));
            }
        }
        if(cnt == 0) System.out.println('0');
    }
}
