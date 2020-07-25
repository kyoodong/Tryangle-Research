package com.company;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;

public class Main {

	public static final String WEB_DRIVER_ID = "webdriver.chrome.driver";
	public static final String WEB_DRIVER_PATH = "chromedriver";
	public static final String IMAGE_DIR = "images";
	public static final int PAGE_NUM = 3000;

	public static void main(String[] args) {
		String template = "https://www.instagram.com/explore/tags/";
		String[] keywordList = {"달달구리", "비오는날", "힐링타임", "화창한", "화창한날씨", "화창한날", "갤러리", "갤러리카페", "그릴",
//		"기찻길", "나무", "남친짤", "디저트스타그램", "럽스타", "루프탑", "루프탑카페", "맥주스타그램", "맥주잔", "맥주한잔", "비오는날", "수영장",
//		"악기", "여친짤", "여행스타그램", "여행에미치다", "여행에미치다_국내", "여행에미치다_제주", "여행에미치다_한국", "와인", "와인바",
//		"와인스타그램", "카페놀이", "카페인테리어", "커플스타그램", "커피스타그램", "케이크", "케이크데코", "케이크디자인", "케이크맛집",
				"케이크주문제작", "케이크클래스", "펍", "펍인테리어", "해변", "해변가", "해변룩", "해변의여인"};

		for (String keyword : keywordList) {
			new Thread(new Runnable() {
				@Override
				public void run() {
					WebDriver webDriver = new ChromeDriver();
					System.setProperty(WEB_DRIVER_ID, WEB_DRIVER_PATH);

					String url = template + keyword;
					String imageDir = IMAGE_DIR + "/" + keyword + "/";

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
							Elements h2List = body.getElementsByTag("h2");
							if (h2List.size() == 0)
								continue;

							Element h2 = h2List.get(1);
							Element recent = h2.parent().child(2);
							Elements images = recent.getElementsByTag("img");
							for (Element image : images) {
								String[] srcList = image.attr("srcset").split("[ ,]");
								if (srcList.length < 2)
									continue;

								String imagePath = srcList[srcList.length - 2];
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
			}).start();
		}
	}
}
