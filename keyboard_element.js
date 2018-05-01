"use strict";
/* Edited by Ryan Stauffer
ryan@enharmonic.ai
*/


/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// exports.__esModule = true;
var offsets = [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6];
var minNote = 60;
var maxNote = 72;
var KeyboardElement = /** @class */ (function () {
    function KeyboardElement(container) {
        this.container = container;
        this.keys = {};
        this.resize();
        this.notes = {};
    }
    KeyboardElement.prototype.resize = function () {
        // clear the previous ones.
        this.keys = {};
        this.container.innerHTML = '';
        // each of the keys.
        var numNotes = maxNote - minNote + 1
        var keyWidth = 1 / numNotes * 1.5;
        for (var i = minNote; i <= maxNote; i++) {
            var key = document.createElement('div');
            key.classList.add('key');
            var isSharp = ([1, 3, 6, 8, 10].indexOf(i % 12) !== -1);
            key.classList.add(isSharp ? 'black' : 'white');
            this.container.appendChild(key);
            // position the element
            var noteOctave = Math.floor(i / 12) - Math.floor(minNote / 12);
            // let offset = (i - minNote) / numNotes
            var offset = offsets[i % 12] + noteOctave * 7;
            key.style.width = keyWidth * 100 + "%";
            key.style.left = offset * keyWidth * 100 + "%";
            key.id = i.toString();
            var fill = document.createElement('div');
            fill.classList.add('fill');
            key.appendChild(fill);
            this.keys[i] = key;
        }
    };
    KeyboardElement.prototype.keyDown = function (noteNum) {
        if (noteNum in this.keys) {
            var key = this.keys[noteNum];
            var note = new Note(key.querySelector('.fill'));
            if (!this.notes[noteNum]) {
                this.notes[noteNum] = [];
            }
            this.notes[noteNum].push(note);
        }
    };
    KeyboardElement.prototype.keyUp = function (noteNum) {
        if (noteNum in this.keys) {
            if (!(this.notes[noteNum] && this.notes[noteNum].length)) {
                console.warn('note off before note on');
            }
            else {
                this.notes[noteNum].shift().noteOff();
            }
        }
    };
    return KeyboardElement;
}());
// exports.KeyboardElement = KeyboardElement;
var Note = /** @class */ (function () {
    function Note(element) {
        this.element = element;
        this.element.classList.add('active');
    }
    Note.prototype.noteOff = function () {
        this.element.classList.remove('active');
    };
    return Note;
}());
