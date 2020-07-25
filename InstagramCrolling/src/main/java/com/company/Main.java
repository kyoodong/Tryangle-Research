package com.company;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.net.URL;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Properties;
import java.util.Queue;

public class Main {

	public static final String WEB_DRIVER_ID = "webdriver.chrome.driver";
	public static final String WEB_DRIVER_PATH = "chromedriver";
	public static final String IMAGE_DIR = "images";
	public static final int PAGE_NUM = 3000;
	public static final int WAIT_TIME = 5000;
	public static final int MAX_NUM_BROWSER = 5;
	int restBrowserNum;

	void start() {
		try {
			Properties properties = new Properties();
			FileInputStream fis = new FileInputStream("instaAccount.properties");
			properties.load(fis);

			String id = properties.getProperty("id");
			String pw = properties.getProperty("pw");
			craw(id, pw);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	void craw(String id, String pw) {
		String template = "https://www.instagram.com/explore/tags/";
		Queue<String> keywordQueue = new ArrayDeque<>();
		String[] keywordList = {
		"기찻길", "나무", "남친짤", "디저트스타그램", "럽스타", "루프탑", "루프탑카페", "맥주스타그램", "맥주잔", "맥주한잔", "비오는날", "수영장",
		"악기", "여친짤", "여행스타그램", "여행에미치다", "여행에미치다_국내", "여행에미치다_제주", "여행에미치다_한국", "와인", "와인바",
		"와인스타그램", "카페놀이", "카페인테리어", "커플스타그램", "커피스타그램", "케이크", "케이크데코", "케이크디자인", "케이크맛집",
				"케이크주문제작", "케이크클래스", "펍", "펍인테리어", "해변", "해변가", "해변룩", "해변의여인"
		};

		keywordQueue.addAll(Arrays.asList(keywordList));
		restBrowserNum = MAX_NUM_BROWSER;

		while (!keywordQueue.isEmpty()) {
			if (restBrowserNum <= 0) {
				try {
					Thread.sleep(1000 * 60);
				} catch (Exception e) {
					e.printStackTrace();
				}
				continue;
			}

			restBrowserNum--;
			String keyword = keywordQueue.poll();
			WebDriver webDriver = new ChromeDriver();
			System.setProperty(WEB_DRIVER_ID, WEB_DRIVER_PATH);

			String url = template + keyword;

			try {
				webDriver.get(url);
				Thread.sleep(WAIT_TIME);

				WebElement loginButton = webDriver.findElement(new By.ByXPath("//button[text()='로그인']"));
				loginButton.click();
				Thread.sleep(WAIT_TIME);

				webDriver.findElement(new By.ByClassName("coreSpriteFacebookIcon")).click();
				Thread.sleep(WAIT_TIME);

				webDriver.findElement(new By.ById("email")).sendKeys(id);
				webDriver.findElement(new By.ById("pass")).sendKeys(pw);
				webDriver.findElement(new By.ById("loginbutton")).click();
				Thread.sleep(WAIT_TIME);

				webDriver.get(url);
				Thread.sleep(WAIT_TIME);
			} catch (Exception e) {
				e.printStackTrace();
			}

			new Thread(new Runnable() {
				@Override
				public void run() {
					String imageDir = IMAGE_DIR + "/" + keyword + "/";

					try {
						Thread.sleep(WAIT_TIME);
						File dir = new File(imageDir);
						if (!dir.exists())
							dir.mkdirs();

						for (int i = 1; i <= PAGE_NUM; i++) {
							if (i % 50 == 0) {
								System.out.println(keyword + ": step " + i);
							}

							Document doc = Jsoup.parse(webDriver.getPageSource());
							Element body = doc.body();
							Elements h2List = body.getElementsByTag("h2");
							if (h2List.size() <= 1)
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
							jse.executeScript("window.scrollBy(0, -window.innerHeight * 2)");
							Thread.sleep(200);
							jse.executeScript("window.scrollBy(0, window.innerHeight * 4)");
							Thread.sleep(100);
						}
					} catch (Exception e) {
						e.printStackTrace();
					} finally {
						webDriver.close();
						System.out.println(keyword + " Done");
						synchronized (this) {
							restBrowserNum++;
						}
					}
				}
			}).start();
		}
	}

	public static void main(String[] args) {
		Main main = new Main();
		main.start();
	}
}
