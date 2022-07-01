import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        try {
            FileInputStream fin = new FileInputStream("./inputstream.tmp");
            int c;
            while((c = fin.read()) != -1) {
                System.out.print((char) c);
            }
            fin.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally {
        }
    }
}
