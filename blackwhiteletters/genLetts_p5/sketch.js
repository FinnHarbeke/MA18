var pics;
var w;
var grid;
var space;
const scale = 2;
var counter;

function setup() {
	w = 32;
	space = 5;
	grid = 10

	createCanvas(((w + space) * grid - space)*scale, ((w + space) * grid - space)*scale);
	pics = [];
	for (let i = 0; i < grid*grid; i++) {
		pics.push(new Pic());
	}
	angleMode(DEGREES);
	counter = 1;
}

function draw() {
	background(100);
	for (let j = 0; j < grid; j++) {
		for (let i = 0; i < grid; i++) {
			pics[j*grid + i].show(i * (w + space), j * (w + space));
		}
	}
	let removed = pics.shift();
	removed.save('pics/' + str(removed.letter) + str(counter))
	pics.push(new Pic())
	counter++;
}

class Pic {
	constructor() {
		this.w = w;
		this.h = w;
		// Gen white background
		this.pixels = [];
		for (let j = 0; j < this.h; j++) {
			this.pixels.push([]);
			for (let i = 0; i < this.w; i++) {
				this.pixels[j].push(1);
			}
		}
		this.scale = floor(random() * w);
		this.rotation = random() * 360;
		// LETTER
		this.letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[floor(random()*26)];
	}

	show(x, y) {
		noStroke();
		fill(0);
		rect(x * scale, y * scale, this.w * scale, this.h * scale);
		// text
		fill(255);
		translate((x + this.w/2) * scale, (y + this.h/2) * scale);
		rotate(this.rotation);

		textSize(this.scale * scale);
		textAlign(CENTER, CENTER);
		text(this.letter, 0, 0);
		
		rotate(-this.rotation);
		translate(-((x + this.w/2) * scale), -((y + this.h/2) * scale));
	}

	save(filename) {
		
	}

}