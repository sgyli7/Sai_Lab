"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const v8_1 = __importDefault(require("v8"));
const electron_1 = require("electron");
const Bootstrap_1 = __importDefault(require("./main/Bootstrap"));
v8_1.default.setFlagsFromString('--harmony');
new Bootstrap_1.default(electron_1.app).start();
//# sourceMappingURL=index.js.map