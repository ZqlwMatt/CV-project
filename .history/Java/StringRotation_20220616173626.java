public class StringRotation {
	public static void main(String[] args) {
		String str1 = "asdasdasd", str2 = "sdasdasda";
		if (str1.length() != str2.length()) {
			System.out.println("-1");
		}
		else {
			str1 = str1.concat(str2);
			// System.out.println(str1);
			if(str1.indexOf(str2) != -1) {
				System.out.println(str1.indexOf(str2));
			else {

			}
		}
	}
}