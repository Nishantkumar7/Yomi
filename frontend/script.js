async function upload() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return alert("Select a PDF first.");

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("output").innerText = data.text || data.error;
}
