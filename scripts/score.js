// Score class
const VF = Vex.Flow;

export class MelodyScore {
  constructor(containerId) {
    this.score = document.getElementById(containerId);
    this.renderer = new VF.Renderer(this.score, VF.Renderer.Backends.SVG);

    // Configure the rendering context.
    this.renderer.resize(330, 130);
    this.context = this.renderer.getContext();
    this.context.setFont('Arial', 10, '')
        .setBackgroundFillStyle('#eed');

    this.stave = this.drawBlankStave();
    this.notes = [];
    this.noteSpacing = 30;
 
    this.tickContext = new VF.TickContext();
    this.tickContext.preFormat().setX(this.noteSpacing);
    this.visibleNoteGroups = [];

    this.lastPitch = {pitchClass: 'c', accidental: '', octave: '4'}; 
  }

  drawBlankStave() {
    // Create a stave and clef
    const stave = new VF.Stave(10, 0, 330).addClef('treble');
    
    // Draw it
    stave.setContext(this.context).draw();
    
    return stave;
  }

  clearScore() {
    d3.select('#score').select('svg').selectAll('*').remove(); 
    this.notes = [];
    this.stave = this.drawBlankStave();
    this.tickContext.preFormat().setX(this.noteSpacing);
    this.lastPitch = {pitchClass: 'c', accidental: '', octave: '4'}; 
    this.visibleNoteGroups = [];
  }

  // Add a note to the VexFlow staff and re-render
  addNoteToStaff(pitchNum) {
    let pitch = pitchNumbertoStaffNote(pitchNum); 
    let note = new VF.StaveNote({
      keys: [pitch.pitchClass + pitch.accidental + '/' + pitch.octave],
      duration: 'q',
      clef: 'treble'
    }).setContext(this.context)
      .setStave(this.stave);

    this.tickContext.addTickable(note);

    if (pitch.accidental) {
      note.addAccidental(0, new VF.Accidental(pitch.accidental));
    } else if (pitch.pitchClass == this.lastPitch.pitchClass &&
               this.lastPitch.accidental) {
      note.addAccidental(0, new VF.Accidental('n'));
    }

    note.preFormat();
    this.notes.push(note);

    const group = this.context.openGroup();
    this.visibleNoteGroups.push(group);
    note.draw();
    this.context.closeGroup();

    this.tickContext.x += this.noteSpacing;

    // Prepare for next pitch addition
    this.lastPitch = pitch;
  }
}

const staffMapping = {
  60: ['c', '', '4'],
  61: ['d', 'b', '4'],
  62: ['d', '', '4'],
  63: ['e', 'b', '4'],
  64: ['e', '', '4'],
  65: ['f', '', '4'],
  66: ['g', 'b', '4'],
  67: ['g', '', '4'],
  68: ['a', 'b', '4'],
  69: ['a', '', '4'],
  70: ['b', 'b', '4'],
  71: ['b', '', '4'],
  72: ['c', '', '5']
} 
 
function pitchNumbertoStaffNote(pitchNum) {
  let array = staffMapping[pitchNum];
  return {
    pitchClass: array[0],
    accidental: array[1],
    octave: array[2]
  };
}


export class TargetScore {
  constructor(containerId) {
    this.score = document.getElementById(containerId);
    this.renderer = new VF.Renderer(this.score, VF.Renderer.Backends.SVG);

    // Configure the rendering context.
    this.renderer.resize(330, 200);
    this.context = this.renderer.getContext();
    this.context.setFont('Arial', 10, '')
        .setBackgroundFillStyle('#eed');

    this.stave = this.drawBlankStave();
    this.notes = [];
    this.noteSpacing = 30;
 
    this.tickContext = new VF.TickContext();
    this.tickContext.preFormat().setX(this.noteSpacing);
    this.visibleNoteGroup = null;

    this.lastPitch = {pitchClass: 'c', accidental: '', octave: '4'}; 
    this.loaded = false;
  }

  drawBlankStave() {
    const stave = new VF.Stave(10, 0, 330).addClef('treble');
    stave.setContext(this.context).draw();
    
    return stave;
  }

  clearScore() {
    d3.select('#score').select('svg').selectAll('*').remove(); 
    this.notes = [];
    this.stave = this.drawBlankStave();
    this.tickContext.preFormat().setX(this.noteSpacing);
    this.lastPitch = {pitchClass: 'c', accidental: '', octave: '4'}; 
    this.visibleNoteGroups = [];
  }

  // Input is a list of target pitch numbers
  addTargetMelodyToStaff(targetNotes) {
    let pitch, note;
    this.notes = [];
    for (let n = 0; n < targetNotes.length; n++) {
      console.log(targetNotes[n] + 60);
      pitch = pitchNumbertoStaffNote(targetNotes[n]+ 60); 
      note = new VF.StaveNote({
        keys:[pitch.pitchClass + pitch.accidental + '/' + pitch.octave],
        duration: 'q',
        clef: 'treble'
      });

      if (pitch.accidental) {
        note.addAccidental(0, new VF.Accidental(pitch.accidental));
      } else if (pitch.pitchClass == this.lastPitch.pitchClass
                 && this.lastPitch.accidental) {
        note.addAccidental(0, new VF.Accidental('n'));
      }

      this.notes.push(note);
      this.lastPitch = pitch;
    }
    console.log(this.notes);
    this.voice = new VF.Voice({
      num_beats: targetNotes.length,
      beat_value: 4
    });
    this.voice.addTickables(this.notes);
    const formatter = new VF.Formatter()
        .joinVoices([this.voice])
        .format([this.voice], 200);
    this.visibleNoteGroup = this.context.openGroup();

    this.voice.draw(this.context, this.stave);
    this.context.closeGroup();
    this.loaded = true;
  }

 recolorNotesFromProbs(noteCorrectProbs) {
    let fillColor, style;
    const notes = [];
    for (let n = 0; n < this.notes.length; n++) {
      fillColor = scaleProbToColor(noteCorrectProbs[n]);
      style = {fillStyle: fillColor};
      if (noteCorrectProbs[n] > 0.98) { 
        style.fillStyle = 'steelblue'; 
      }
      this.notes[n].setStyle(style);
      notes.push(this.notes[n]);
    }

    // Clear the existing notes
    this.context.svg.removeChild(this.visibleNoteGroup);

    const voice = new VF.Voice({num_beats: notes.length, beat_value: 4});
    voice.addTickables(notes);
    const formatter = new VF.Formatter()
        .joinVoices([voice])
        .format([voice], 200);
    this.visibleNoteGroup = this.context.openGroup();

    voice.draw(this.context, this.stave);
    //this.voice.draw(this.context, this.stave);
    this.context.closeGroup();
  }
}

const scaleProbToColor = d3.scaleLinear()
    .range(['black', '#579848'])
    .domain([0, 1]);
