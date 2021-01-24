#coding:utf-8
import time
from PIL import Image, ImageTk
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from retry import retry

# fill your id and password here
user = ''
pwd = ''
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
browser = webdriver.Chrome(chrome_options=chrome_options)
wait = WebDriverWait(browser,10)
browser.implicitly_wait(10)
browser.maximize_window()
on_hit = False

@retry()
def browser_click(browser, xpath):
    wait = WebDriverWait(browser, 1)
    wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    browser.find_element(By.XPATH, xpath).click()
    return None

@retry()
def browse_log():
    browser.implicitly_wait(10)
    browser.get('http://portal.pku.edu.cn')
    print('begin to connect')
    #browser.find_element(By.LINK_TEXT, u"您好，请登录").click()
    #wait.until(EC.presence_of_element_located(By.NAME, 'userName'))
    username = browser.find_element(By.NAME, 'userName')
    username.send_keys(user)
    password = browser.find_element(By.NAME, 'password')
    password.send_keys(pwd)
    browser.find_element(By.ID, 'logon_button').click()
    wait.until(EC.element_to_be_clickable((By.ID, 'epidemic')))
    browser.find_element(By.ID, 'epidemic').click()
    handles = browser.window_handles
    browser.switch_to.window(handles[-1])
    print('success connect')

    browser_click(browser, '//*[@id="pane-daily_info_tab"]/form/div[13]/div/label[2]')
    browser_click(browser, '//*[@id="pane-daily_info_tab"]/form/div[14]/div/div/div[1]/span/span/i')
    time.sleep(3)
    browser_click(browser, '/html/body/div[2]/div[1]/div[1]/ul/li[1]')
    browser.save_screenshot('./success.png')
    browser_click(browser, '//*[@id="pane-daily_info_tab"]/form/div[17]/div/button')
    img = Image.open('./success.png')
    img.show()
    print('end connect')
    return None

if __name__ == "__main__":
    browse_log()
    exit()