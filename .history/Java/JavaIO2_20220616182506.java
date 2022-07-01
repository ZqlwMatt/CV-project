import java.io.*;
public class JavaIO2 {
    public static void main(String[] args) {
        try {
            FileReader fr = new FileReader("./inputstream.tmp");
            FileWriter fw = new FileWriter("./outputstream.tmp");
            int c;
            while((c = fr.read()) != -1) {
                System.out.print((char) c);
                fw.write((char) c);
            }
            fr.close();
            fw.close();

        }
        catch(IOException e) {
            e.printStackTrace();
            System.out.println(e);
        }
        finally {

        }
    }
}
