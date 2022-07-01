import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        File file = new File("JavaIO.test");
        if(file.createNewFile()) {
            System.out.println("1");
        }
        else {
            System.out.println("0");
        }
    }
}
