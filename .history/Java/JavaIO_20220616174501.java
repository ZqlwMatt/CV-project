import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        File file = new File("JavaIO.test");
        try {
            if(file.createNewFile()) {
                System.out.println(1);
            }
            else {
                System.out.println(0);
            }
        }
        catch(IOException e) {

        }
        finally {
            System.out.println("Finally?");
        }
    }
}
