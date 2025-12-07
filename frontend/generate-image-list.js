import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const normalDir = path.join(__dirname, 'public', 'test', 'NORMAL');
const pneumoniaDir = path.join(__dirname, 'public', 'test', 'PNEUMONIA');

const normalImages = fs.existsSync(normalDir)
    ? fs.readdirSync(normalDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    : [];

const pneumoniaImages = fs.existsSync(pneumoniaDir)
    ? fs.readdirSync(pneumoniaDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    : [];

console.log('='.repeat(60));
console.log('Found Images:');
console.log('='.repeat(60));
console.log(`NORMAL: ${normalImages.length} images`);
console.log(`PNEUMONIA: ${pneumoniaImages.length} images`);
console.log('='.repeat(60));

console.log('\n// Copy this into App.jsx (replace lines 18-31):\n');

console.log('const normalImages = [');
normalImages.forEach(img => {
    console.log(`  '${img}',`);
});
console.log(']');

console.log('\nconst pneumoniaImages = [');
pneumoniaImages.forEach(img => {
    console.log(`  '${img}',`);
});
console.log(']');

console.log('\n' + '='.repeat(60));
console.log(`Total: ${normalImages.length + pneumoniaImages.length} images will be displayed`);
console.log('='.repeat(60));
