document.addEventListener("DOMContentLoaded", function() {
    const animatedElements = document.querySelectorAll('.animated');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, {
        threshold: 0.5
    });

    animatedElements.forEach(element => {
        observer.observe(element);
    });
});
