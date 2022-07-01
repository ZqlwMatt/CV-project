import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        File file = new File("JavaIO.test");
        try {
            File file2 = new File("JavaIO2.test");
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