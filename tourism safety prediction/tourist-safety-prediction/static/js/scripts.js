// Basic form validation
document.querySelector('form')?.addEventListener('submit', function(e) {
    const inputs = document.querySelectorAll('select');
    for (let input of inputs) {
        if (!input.value) {
            e.preventDefault();
            alert('Please fill all fields');
            break;
        }
    }
});