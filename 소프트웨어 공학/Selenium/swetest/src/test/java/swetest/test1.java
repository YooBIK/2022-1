package swetest;

import java.time.Duration;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.openqa.selenium.By;
import org.openqa.selenium.Keys;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

import io.github.bonigarcia.wdm.WebDriverManager;

public class test1 {
	WebDriver driver;
	
	@Before
	public void init() {
		WebDriverManager.chromedriver().setup();
		driver = new ChromeDriver();
		driver.manage().timeouts().implicitlyWait(Duration.ofSeconds(20));

		driver.manage().timeouts().scriptTimeout(Duration.ofMinutes(2));

		driver.manage().timeouts().pageLoadTimeout(Duration.ofSeconds(20));
	}
	
	@Test
	public void openBrowser() {
		driver.get("https://www.amazon.com/");
		driver.findElement(By.name("field-keywords")).sendKeys("samsungssd");
		driver.findElement(By.id("nav-search-submit-button")).click();

	}
	
	@After
	public void finish() {
		//driver.close(); //
	}
}
