import java.io.*;
public class JavaIO {
    public static void main(String[] args) {
        File file = new File("JavaIO.test"); // ???
        File file2 = new File("JavaIO2.test");
        try {
            if(file2.createNewFile()) { // 文件在这里创建
                System.out.println(1);
                System.out.println(file); // 打印 FileName

                file = file2.getCanonicalFile();
                String path = file.getAbsolutePath();
                System
            }
            else {
                System.out.println(0);
                System.out.println(file2.exists());y
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