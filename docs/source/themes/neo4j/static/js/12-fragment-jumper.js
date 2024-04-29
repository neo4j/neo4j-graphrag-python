;(function () {
  'use strict'

  var article = document.querySelector('article.doc')
  var cheatSheet = document.querySelector('body.cheat-sheet')
  var toolbar = document.querySelector('.toolbar')
  var headerNavigationBar = document.querySelector('header > .navbar')

  function decodeFragment (hash) {
    return hash && (~hash.indexOf('%') ? decodeURIComponent(hash) : hash).slice(1)
  }

  function computePosition (el, sum) {
    if (article.contains(el)) {
      return computePosition(el.offsetParent, el.offsetTop + sum)
    } else {
      return sum
    }
  }

  function jumpToAnchor (e) {
    if (e) {
      window.location.hash = '#' + this.id
      e.preventDefault()
    }
    var topOffset = toolbar ? toolbar.getBoundingClientRect().bottom : headerNavigationBar.getBoundingClientRect().bottom

    if (cheatSheet) {
      var scrollTarget = this.closest('div')
      var selectorsTop = document.querySelector('.nav-container .selectors').querySelector('div').getBoundingClientRect().top
      if (this.tagName === 'H3') topOffset = selectorsTop
      window.scrollTo(0, computePosition(scrollTarget, 0) - topOffset)
    } else {
      window.scrollTo(0, computePosition(this, 0) - topOffset)
    }
  }

  window.addEventListener('load', function jumpOnLoad (e) {
    var fragment, target
    if ((fragment = decodeFragment(window.location.hash)) && (target = document.getElementById(fragment))) {
      jumpToAnchor.bind(target)()
      setTimeout(jumpToAnchor.bind(target), 0)
    }
    window.removeEventListener('load', jumpOnLoad)
  })

  Array.prototype.slice.call(document.querySelectorAll('a[href^="#"]')).forEach(function (el) {
    var fragment, target
    if ((fragment = decodeFragment(el.hash)) && (target = document.getElementById(fragment))) {
      el.addEventListener('click', jumpToAnchor.bind(target))
    }
  })
})()
