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
            byte b[] = "outputstream".getBytes(); // 转换成二进制打印 有点蠢
            fout.write(b);
            fout.close();

        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally {

        }
    }
}
