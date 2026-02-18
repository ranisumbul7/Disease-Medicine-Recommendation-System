function sendMessage() {
    const message = document.getElementById("userInput").value;

    fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("response").innerText = data.reply;
    });
}
