import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        try {
            FileInputStream fin = new FileInputStream("./inputstream.tmp");
            int c;
            while((c = fin.read()) != -1) {
                System.out.print((char) c); // 逐个字符打印
            }
            fin.close();

            FileOutputStream fout = new FileOutputStream("./outputstream.tmp");
            fout.write(65);
            fout.write("outputstream")
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally {

        }
    }
}
