import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        File file = new File("JavaIO.test");
        try {
            File file = new File("JavaIO.test");
            if(file.createNewFile()) { // 文件在这里创建
                System.out.println(1);
            }
            else {
                System.out.println(0);
            }
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        finally {
            System.out.println("finally");
        }
    }
}

// file.createNewfFile()
// catch(IOException e)
// e.printStackTrace();
// finally