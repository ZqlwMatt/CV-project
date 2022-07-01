import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        try {
            FileInputStream fin = new FileInputStream("./inputstream.tmp");
            int a = fin.read();
            System.out.print((char) a);
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally [
            
        ]
    }
}
