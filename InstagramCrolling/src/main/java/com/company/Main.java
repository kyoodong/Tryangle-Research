package com.company;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;

public class Main {

	public static final String WEB_DRIVER_ID = "webdriver.chrome.driver";
	public static final String WEB_DRIVER_PATH = "chromedriver";
	public static final String IMAGE_DIR = "images";
	public static final int PAGE_NUM = 5000;

    public static void main(String[] args) {
	    String template = "https://www.instagram.com/explore/tags/";
	    String[] keywordList = {"카페인테리어"};

//		try {
//			webDriver.get("https://www.instagram.com/?hl=ko");
//			Thread.sleep(1000);
//			webDriver.findElement(new By.ByClassName("coreSpriteFacebookIcon")).click();
//			Thread.sleep(1000);
//
//			webDriver.findElement(new By.ById("email")).sendKeys("아이디");
//			webDriver.findElement(new By.ById("pass")).sendKeys("패스워드");
//			webDriver.findElement(new By.ById("loginbutton")).click();
//			Thread.sleep(1000);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}

	    for (String keyword : keywordList) {
			String url = template + keyword;
			String imageDir = IMAGE_DIR + "/" + keyword + "/";

			ChromeOptions options = new ChromeOptions();
			WebDriver webDriver = new ChromeDriver();
			System.setProperty(WEB_DRIVER_ID, WEB_DRIVER_PATH);

			try {
				File dir = new File(imageDir);
				if (!dir.exists())
					dir.mkdirs();

				// webdriver 방식
				webDriver.get(url);
				Thread.sleep(4000);
				for (int i = 0; i < PAGE_NUM; i++) {
					System.out.println(keyword + ": step " + i);

					Document doc = Jsoup.parse(webDriver.getPageSource());
					Element body = doc.body();
					Element h2 = body.getElementsByTag("h2").get(1);
					Element recent = h2.parent().child(2);
					Elements images = recent.getElementsByTag("img");
					for (Element image : images) {
						String imagePath = image.attr("srcset").split(" ")[0];
						String[] names = imagePath.split("/");
						String filename = null;
						for (String name : names) {
							if (name.contains(".jpg")) {
								filename = name.substring(0, name.indexOf(".jpg") + 4);
								break;
							}
						}

						if (filename == null)
							continue;

						BufferedImage bufferedImage = ImageIO.read(new URL(imagePath));
						File file = new File(imageDir + filename);
						if (file.exists()) {
							continue;
						}
						ImageIO.write(bufferedImage, "jpeg", file);
					}

					JavascriptExecutor jse = (JavascriptExecutor) webDriver;
					jse.executeScript("window.scrollBy(0, 2000)");
					Thread.sleep(100);
					jse.executeScript("window.scrollBy(0, 2000)");
					Thread.sleep(100);
				}
			} catch (Exception e) {
				e.printStackTrace();
			} finally {
				webDriver.close();
			}
		}
    }
}
