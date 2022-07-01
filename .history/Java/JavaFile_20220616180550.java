import java.io.*;
public class JavaFile {
    public static void main(String[] args) {
        File file = new File("JavaFile.tmp"); // ???
        File file2 = new File("JavaFile2.tmp");
        try {
            if(file2.createNewFile()) { // 文件在这里创建
                System.out.println(1);

                file = file2.getCanonicalFile();
                System.out.println(file); // 打印新文件的 File Path
                String path = file.getAbsolutePath();
                System.out.println(path); // 打印新文件的 File Path
            }
            else {
                System.out.println(0);
                System.out.println(file2.exists());

                File f = new File("./"); // 获取当前文件夹
                String filenames[] = f.list();
                for(String filename : filenames) {
                    System.out.println(filename); // 打印所有文件名
                }
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