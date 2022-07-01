import java.io.*;
public class JavaIO2 {
    public static void main(String[] args) {
        try {
            FileReader fr = new FileReader("./inputstream.tmp");
            int c;
            while((c = fr.read()) != -1) {
                System.out.print((char) c);
            }
            fr.close();

            FileWriter fw = new FileWriter("");
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally {

        }
    }
}
