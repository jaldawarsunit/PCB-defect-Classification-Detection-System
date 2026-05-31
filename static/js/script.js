function showSection(id){

    document.querySelectorAll(".section").forEach(sec=>{
        sec.classList.remove("active")
    })

    document.getElementById(id).classList.add("active")
}


// Prediction API
document.getElementById("predictForm").onsubmit = async function(e){

    e.preventDefault()

    const formData = new FormData(this)

    const res = await fetch("/predict",{
        method:"POST",
        body:formData
    })

    const blob = await res.blob()

    document.getElementById("finalResult").src = URL.createObjectURL(blob)

    // Show download card
    document.getElementById("downloadCard").style.display = "block"

    // Fetch and display prediction log
    const logRes = await fetch("/get_log")
    const logData = await logRes.json()

    const logBody = document.getElementById("logBody")
    logBody.innerHTML = ""

    if (logData.length === 0) {
        logBody.innerHTML = "<tr><td colspan='7'>No defects detected.</td></tr>"
    } else {
        logData.forEach((row, i) => {
            logBody.innerHTML += `
                <tr>
                    <td>${i + 1}</td>
                    <td>${row[0]}</td>
                    <td>${row[1]}</td>
                    <td>${row[2]}</td>
                    <td>${row[3]}</td>
                    <td>${row[4]}</td>
                    <td>${row[5]}</td>
                </tr>`
        })
    }

    document.getElementById("logCard").style.display = "block"
}