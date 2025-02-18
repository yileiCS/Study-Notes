# JavaScript Study Notes

## RESOURCES
- [Suggest to read before study: A first splash into JavaScript](https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Scripting/A_first_splash) ➡️ this can help to have a quick look about related terminologies and style of this programming language
- [MDN beginner's JavaScript course](https://developer.mozilla.org/en-US/docs/Learn_web_development/Core/Scripting)
- [What you can achieve after learning JavaScript: write awesome jsgame!](https://github.com/proyecto26/awesome-jsgames?tab=readme-ov-file)

---
## Study/Practice Websites
- [W3 Schools](https://www.w3schools.com/js/default.asp)

---
### ESSENTIALS
To add all JavaScript code into HTML by using `<script>` element
  ```
  <script>
    // Your JavaScript goes here
  </script>
  ```
**JavaScript Can Change HTML Content**
➡️
- JavaScript has the ability to dynamically modify the content of an HTML page. 
- After the page is loaded, JavaScript can be used to change the text, style, structure of elements, and more on the page.
  
**`getElementById()`**
- is one of many methods in the HTML DOM (Document Object Model)
- its primary function is to find and return and HTML element based on its 'id' attribute
- commonly used to locate a specific element on the page so that you can manipulate it (e.g., change its content, style, attributes, etc.).
- **Syntax**
  ```
  document.getElementById("element_id")
  ```
  e.g.
    ```
    document.getElementById("demo").innerHTML = "Hello JavaScript";
    ```
   - `getElementById("demo")`
     - This part of code finds the HTML element with `id="demo"` （通过`id="demo"`找到HTML文档中对应的元素
     - if the HTML contains `<div id="demo">Hello World!</div>`）
     - `document.getElementById("demo")` will return the DOM object of the `<div>` element 会返回`<div>`元素的DOM对象
   - **`.innerHTML`**
     - is a property of a DOM element that represents the HTML content inside the element
     - By using the assignment operator (`=`), you can set new content as the value of `innerHTML`
     - e.g. `element.innerHTML = "New Content";`  This will change the inner HTML of `element` to `"New Content"`.


