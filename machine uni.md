
# Connection machine  aux machines Alan GPU Cluster:

## 1. Connection VPN Unif

## 2 . Connection SSH



<!-- The text field -->
- <input type="text" value="ssh phockers@10.19.99.66" id="myInput">
<!-- The button used to copy the text -->
<button onclick="myFunction()">Copy </button>

- <input type="text" value="pyx9140" id="myInput">
<!-- The button used to copy the text -->
<button onclick="myFunction()">Copy text</button>

function myFunction() {
  /* Get the text field */
  var copyText = document.getElementById("myInput");

  /* Select the text field */
  copyText.select();
  copyText.setSelectionRange(0, 99999); /*For mobile devices*/

  /* Copy the text inside the text field */
  document.execCommand("copy");

  /* Alert the copied text */
  alert("Copied the text: " + copyText.value);
}

## change Password
passwd

## Documentation
- [alan-cluster](https://github.com/montefiore-ai/alan-cluster)
- [slurm](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html.)
